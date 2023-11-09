import pytest


@pytest.fixture
def transcript_text():
    return """This session will be all about an extended example program dealing with discrete event simulation of 
    digital circuits. This example will show interesting ways how one can combine assignments and higher order 
    functions. Our task will be to construct a digital circuit simulator. There are a number of things that we need 
    to do. We need to define a way to specify a digital circuit. We need to define a way to how to run simulations 
    and then we have to put everything together so that the circuits can actually drive the simulator. This is also a 
    great example that shows how to build programs that do discrete event simulation. So we start with a small 
    description language for digital circuits. A digital circuit is composed of wires and the functional components. 
    Wires transport signals and components transform signals. We represent signals using Booleans True and False. So 
    it's a digital circuit simulator and analog one. A signal is either true or false and not but not something in 
    between. Their base components are also called gates. They are the inverter whose output is the inverse of its 
    input. The AND gate whose output is the conjunction, logical AND of its inputs and the OR gate whose output is 
    the disjunction, logical OR of its inputs. Once we have these three, we can construct the other components by 
    combining the base components. One thing important for the simulation is that the components have a reaction time 
    or a delay. That means their outputs don't change immediately after a change in their inputs. Logical gates are 
    usually drawn with standard symbols. So this would be an AND gate. This would be an OR gate and this would be an 
    inverter. We can use the basic gates to build more complicated structures. For instance, here we have created a 
    half-adder. The half-adder takes two inputs, A and B and then it has two outputs, the sum and the carry. The 
    carry is set if both A and B are set. So we have a logical AND gate that goes into the carry output. The sum is 
    set if either A or B is set. We have an OR gate here and the carry is not set. So we have an inverter here. So 
    that one is set if the carry is false and here we have an AND gate. So the sum is true only if either A or B is 
    true and carry is false. Once we have defined a circuit like a half-adder, we can use it in turn to define more 
    complicated circuits. For instance, we can use two half-adders to define a full-adder. So what we do here is we 
    first have a half-adder that adds B and C in, producing a carry here and a sum here. We take the sum and add A, 
    so with another half-adder. So the final sum is the sum of the whole full-adder and the output carry is then the 
    logical OR of the carries of the two half-adders that you see here. So our next question is how do we put that in 
    code? What we have to do is essentially find code representations for these three basic circuit diagrams so that 
    they can be composed to give you half-adders, full-adders and any other components you might wish to construct. 
    So our basic language then consists of the three gates here, plus the wires. So a wire is needed to connect gates 
    and here should be noted that a wire is not, that doesn't necessarily have two ends, essentially any network that 
    has the same current electrically counts as a wire. So for instance the A wire would be not just this line here, 
    but it would also that would be part of the A wire and the C wire would consequently be the whole network that 
    you see here. So these colored things are wires and then we have gates that connect the wires, that's the basic 
    situation. You see also that these circuits are not necessarily hierarchical, they're potentially complex 
    networks of components connected by wires. The way we're going to represent that is that we are going to have a 
    class for wires and then we have a model where essentially we drop a gate, a component onto a circuit board 
    connecting it with the wires that are there. So essentially our idea to construct a circuit is based on side 
    effects. We can think of it as, essentially you have a circuit board, you have some wires and then essentially 
    you put the gates as an action onto this board and connect them with the wires. That's the mental model that 
    we're going to pursue. So to start we need a class wire to model wires. So wires can then be constructed as 
    follows, we can say A equals a new wire, B is a new wire, C is a new wire or a scalar allows you to shorten that, 
    you can just write well A, B, C equals wire. That means each of these three names will get the same right hand 
    side, namely create a new wire. And then we need functions that create the base components as a side effect. We 
    drop them on the board. So inverter says place an inverter between the input wire here and the output wire there. 
    And gate says place an AND gate between the two inputs here and the output there. And ORGAD says create an ORGAD 
    with those two inputs and the output. All three methods have unit as the result type. That means they act only as 
    a side effect by essentially connecting themselves to the wires. They don't return anything interesting. Starting 
    with these basic elements, wires and three kinds of gates, we can construct more complex components. For 
    instance, here's how we can define a half adder. So it says we have four wires, a V, S and C. We create some 
    internal wires which we're going to use later. We place an ORGAD between AB and D. So that would be an ORGAD here 
    and that would be D. We place an AND gate between AB and C. So that would be my AND gate here and that would be 
    C. I'm connected already. We place an inverter between C and E. So that would be E now. And we place an AND gate 
    between DE and F. Okay, and that's an AND gate, not an ORGAD. Okay, and that's my half adder. So I define a 
    method which means I draw a box around it. And now I can use this thing as it's separate components. So now I can 
    start to drop half adders on my circuit board and connect them to further wires. So here's how you would do a 
    full adder. It takes three input wires, two output wires, S3 internal wires, and you place two half adders on the 
    board connected as you see here. And an ORGAD for the two output carries that gives you the final C out signal. I 
    invite you to do that yourself for just following these instructions and verifying that you will get the full 
    adder that we've seen some slides ago. So here's an exercise for you. What logical function does this program 
    describe? There's a function F, a mystery function that takes two input wires, A and B, and has an output wire C. 
    And here are its internal workings. What's the logical function that gives? Is it one of these six that you see 
    here below? So one way to solve this puzzle is to look at these gates as describing logical formulas. For 
    instance, this inverter here would define the signal D to be not A. Similarly, E equals not B. Then F is A and E. 
    So F is A and not B. G is B and not A. And the final result C is F or G. So the final result is true if either A 
    is true in B is false or B is true and A is false. Which means to say that the final result is true if A and B 
    are different signals. So that's the answer to our puzzle. Okay, let's proceed to an implementation of all this. 
    So the class wire and the functions inverter, Amgate and Orgate represent a small description language for 
    digital circuits. We now need an implementation of this class and these methods to allow us to simulate circuits. 
    So to be able to do this, we first have to clarify how do we do simulation? What interface do we have to run a 
    simulation? So what we do next is develop a simple API for discrete event simulation. So a discrete event 
    simulator performs actions that are specified by the user to be performed at a given moment. And an action is a 
    function that doesn't take any parameters and which returns unit. So an action is really just something that 
    lives for the action, the side effect that it performs. And the time when an action is performed is simulated. It 
    has nothing to do with the actual time. So there's an internal timekeeping unit that essentially advances the 
    time as the simulation progresses. So the way we'll set it up is that a concrete simulation will be inside an 
    object that inherits from the trait simulation. And the trait simulation has the following signature of methods 
    and members that we can use in our simulation object. So first there's the current time which is the simulated 
    time. So that's the current time in our simulation. Then there's a method after delay that registers an action to 
    be performed after a certain delay relative to the current time. So we say after delay and units run this block 
    of statements. So what this does is it essentially stores this block to be run. Once the simulation has reached 
    the time that's specified by current time plus delay. But after delay doesn't run the simulation by itself that's 
    done by a separate method called run. So run essentially says well the actions that are now stored for the 
    simulation they should be performed at this point. Before going into details I want to show you the outline of 
    the different components that we are going to assemble. So at the very top is the simulation trait and that will 
    be inherited by essentially our structure that defines the basic gates. Gates need simulation because it's in the 
    end them that will be simulated. And then from gates we will have another trait called circuits that contains 
    things like half-adders, full-adders or other circuits that a user might want to define. So here we sort of 
    appear that essentially the system provides a simulation package and it provides the basic gates. When we define 
    circuits then maybe some of them are provided by the system and others could be provided by the user. So we could 
    also have several classes here that essentially define different libraries of circuits. In our example we just 
    need a single one. And then finally we will have an object called it simulator which is the concrete simulation. 
    So the concrete simulation essentially defines a test circuit that we now want to run and simulate. So we have 
    already shown the circuits layer when we defined half-adders and full-adders. So that showed how to define these 
    circuits. What we still need to do is we need to fill in the blanks for the gates class. What do inverters and 
    end-gate or gate actually do and how is why I defined and we have to fill in the simulation class to implement 
    the API that we have defined. So let's turn next to gates and in gates let's turn to wire. So a wire must support 
    three basic operations. The simulation might want to know what is the current signal on the wire. Is it true or 
    false? It might want to set the signal of a wire as an action so that would modify the value of the signal 
    transported by the wire. And the third basic operation add action allows the simulation to customize what should 
    happen when the signal of a wire changes. We can add an action, we can attach that to the actions of a wire and 
    all of these attached actions are executed at each change of the transported signal. So it basically we register 
    an action to say perform this action when signal of the wire changes. So here's an implementation of class wire. 
    Internally a wire would have a signal value, initially it's false, so no current and that's private so you can't 
    access it from the outside. And the other piece of state in a wire is the list of actions that are currently 
    attached to the wire. Initially that's the empty list. Now here are the three methods. Get signals simply returns 
    the current signal value. So the set signal operation sets the current signal value to the signal s and it also 
    executes all the stored actions if the signal value changes. So this line here it's maybe a bit cryptic so let's 
    analyze what this is. It does a for each and all the actions and what does it do? Well it calls the action with 
    the empty parameter list. So that's what that does. You could write this also a bit more expensive to say for a 
    taken from actions to execute the action a. That's probably a bit clearer. The last operation add action simply 
    adds to given action to the list of actions on the front and then it immediately calls the action. So why is 
    that? Well you can think of the circuit initially to be in an undefined state. So when I said signal value equals 
    false I really should have said undefined. So once we have an action essentially we immediately execute it to 
    essentially force the signal to be defined namely to be the output value of that action. So once we have wires we 
    can now proceed to define the basic gates. So let's start with inverter. We implement an inverter by installing 
    an action on its input wire. So we have here input add action invert action and what does invert action do? Wert 
    will sample the signal on the input wire and set the output to be the negation the knot of the input signal. 
    That's what inversion means. But it will do so not immediately but only after the inverter delay that we still 
    have to specify. So to summarize we have said before that in placing an inverter on the board essentially 
    produces a side effect what the side effect is to add the invert action to the set of actions on the input wire. 
    If we now look at AND gate it's implemented in a very similar way. So again the AND gate function here has a side 
    effect and the side effect is to add the AND action on its two input wires in one and in two. So what is AND 
    action? Well we get the signal from the two input wires here and here and then we set the output signal to be the 
    AND of the two input signals. And the two are only after a delay namely AND gate delay. So this action will be 
    registered in this simulation to be take place at a point in time AND gate delay from the current time that is 
    now. And the OR gate is now implemented quite analogously. So again it adds an action to its input wires that's 
    the OR action and the OR action that simply would take the disjunction, the OR of the two input signals and do so 
    after OR gate delay. So here's a question to you to see whether you follow. What would happen if we compute in 
    one sec and in two sec directly inside the after delay as you see here. So I don't bother to define them as two 
    volts in front. I just essentially inline them here and here. As you see there would that give us the same 
    behavior or would the behavior be different in this case OR gate two would not model the OR gate faithfully. And 
    the answer is of course that would be something different. So here we get the signal at the current simulated 
    time. Let's just to get signal and then we wait OR gate delays and then we set the output signal. Whereas in the 
    modified program we again wait OR gate delay units but we set the output signal to the disjunction of the input 
    signals at this point in the future. And of course at this point in the future something my else might already 
    have happened. So this is not a faithful model of an OR gate. So let's see where we are in the worksheet. So I 
    have here my simulation trait which is still empty. I have my gates trait which is again empty. I just have the 
    class wire and the three methods and I have the delay that are also defined here. And finally I have circuits 
    that are already added, half-edder and full-edder. I just need the interface of the gates class and that 
    interface is provided just the implementation that is still missing. So what we've done then is we have 
    implemented the class wire as you see here. We have implemented the inverter as you see here and we have 
    implemented AND gate and OR gate. So what we have to do next is flesh out the simulation trait. So the idea is to 
    keep in every instance of the simulation trait an agenda of actions to perform. An agenda is simply a list of 
    events and each event is composed of an action and the time when the action must be executed must be produced. 
    The time is simulated time as we said before. The agenda is sorted in such a way that the actions to be performed 
    first are in the beginning of the agenda. So agenda is a list of events and here is our agenda which is initially 
    in NIL. There's also a private variable called current time or code time that contains the current simulation 
    time. An application of the after delay method with some delay and some block given inserts the task event to be 
    produced at current time plus delay to consist of the actions in block into the agenda at the right position. So 
    here's after delay we produce the right event here and we're inserted into the agenda. Inversion function is 
    straightforward so that's essentially just what we did in when we did sorting as well. So we just go through the 
    agenda if the time of the first item in the agenda is less or equal to the time of the item that we want to 
    insert and we insert in the tail of the agenda and otherwise we put the item at the top of the agenda and follow 
    it with the previous agenda. So once we have an agenda we need to execute it that's done in an event handing 
    loop. The event handing loop removes successive elements from the agenda and performs the associated actions. So 
    here's an implementation of the event loop. It does a paramatch on agenda. As long as the agenda is non-empty it 
    executes the first item on the agenda and it recursively calls itself. If the agenda is empty it terminates. What 
    does it mean to execute the first item of the agenda? Well we strip off the item from the agenda so the agenda 
    now becomes the rest here. It sets the current time to the time stored in the first entry and it performs the 
    action of the first event at this time at this simulated time. Quick check whether the loop function is tail 
    recursive. Yes indeed it calls itself as last action. That's important because of course the agenda for 
    simulations might become quite long. So now finally the run method. Run method simply calls loop after it prints 
    essentially ahead of it. Installs the first action after delay zero that says simulation has started and here's 
    the current time. So it puts that at the front of the agenda and executes loop. One question for you does every 
    simulation terminate after the finite number of steps? At first glance it might seem so because since loop just 
    goes through the agenda left to right until the agenda is niddle. However remember that actions that I performed 
    here can install further actions in the agenda through after delay and that means that in fact we might never 
    finish the simulation because every action will install one or more further actions in the agenda and the agenda 
    could even grow without bounds if every action installs more than one action into the agenda. So time to try it 
    out but before we can launch the simulation we still need a way to examine the changes of the signals on the 
    wires. So far it's a black box something happens but we don't know what. So to this end we define another 
    function called probe. So probe is essentially you have a wire and then you have a pair of pliers and very bad 
    drawing and then an old fashioned oscilloscope or something like that that would tell you what goes on in these 
    wires. So that's what a probe is. So a probe gets attached to this wire here and it has a name because we are 
    actually going to print out the signal of the wire just rather than showing it as a curve like here. So what we 
    want to print is the probe action says print the name of the wire, print the current time and print the current 
    value of that wire. And every time the signal on the wire changes this probe action will be executed because we 
    have added probe action as an action to that wire. I have added all these implementations to the worksheet. 
    Here's probe so that's the last thing I've added here. So it's time to set up a simulation. To set up a 
    simulation we define an object, sim and that extends essentially the circuits that we want to simulate. And we 
    get an error and says okay so there are three things that I haven't defined yet. Indeed those were the methods in 
    virtual delay, ANDK delay and ORGET delay. I could define them right here but that wouldn't be very systematic 
    because when we define a library of gates then we don't really want to fix these delays at this point. The delays 
    of these gates is technology dependent. With a new generation of silicon it might be different. So we want to 
    define them somewhere else further towards the actual simulation class. So what we do instead is we create a 
    separate trait for the delays which you see here. So essentially our simulation object extends circuits and 
    delays so it gets the circuits from one part and the delays from another. And here are the delays that I have to 
    find just to do an example. So if I look at my class diagram again I have the classes simulation, gates, 
    circuits and my concrete simulation and the concrete simulation now also inherits from a trait that fixes the 
    technology dependent delays. So here's a sample simulation that we're going to do in the worksheet. We define 
    four wires and place some probes so two input wires are some in a carry wire and we want to place the probes on 
    the sum and the carry. Next we want to define a half-adder using these wires. So he plays a half-adder between 
    input one and put two sum and carry. Now we want to give the value true to input one and launch the simulation. 
    So let's set up the simulation like this. I have the wires, I have the probes and the half-adder in order to do 
    this without prefixing I have just imported the simulation object so that way I have access to everything that's 
    defined in here. So we get the initial values of sum and carry which are both zero. Now to do something let's 
    change the symbol of one of the inputs and run the simulation. So it says simulation started and not more but if 
    we hover over it then we will see okay the sum probe gave us at time eight a new value true. So after eight 
    simulated units the sum signal went to true. What we could do now is we could also set a signal for input two and 
    run again. And what we see now is that the carry and some signals have changed. The carry signal at a simulated 
    time 11 became true and the sum signal at simulated time 16 became false again. So that just shows that the basic 
    of simulations work as expected. I invite you to define more circuits and have more simulation runs to play with 
    it. So in fact logically speaking we wouldn't have needed three gates since for instance the orgate can be 
    defined in terms of and or in. So an alternative for the orgate would be to define it as a circuit that would 
    then correspond to to this circuit here where we first put an inverter here and then inverter there. And then we 
    have an AND gate of the negated symbols and then we put a final inverter on the result wire. That's of course 
    just a consequence of the formula that A or B is the same as not not A and not B. So that's the circuit that we 
    have drawn here. So a question to you what would change in the circuit simulation if the implementation of orgate 
    out that you have just seen was used for or would it be nothing the two simulations behave the same or would the 
    simulations produce the same events but the indicated times are different or would the times be different and 
    orgate out made us might also produce additional events or would the two simulations produce different events are 
    together what do you think. So clearly the timings would be different in general. If you take our current example 
    values then an inverted delay would be two and an AND gate delay would be three. So you get would get a total 
    delay of seven whereas an orgate delay in our example values had a delay of five. So that would give you 
    different times. And you might think that's it. So it would be number two. But if you have actually tried this 
    and you actually will actually see something else the times are different and orgate out might also produce 
    additional events. So to demonstrate I have plugged in the new version of the orgate in the worksheet. And now we 
    have the run here where we get for the first run at time at time number five the value is true and then at 10 the 
    value is first false and then true again. So that looks at first really mysterious. To explain the mystery let's 
    have another look at this diagram again. So what we have here is essentially not a single event as in the orgate 
    where essentially we have a single action to be produced but we have multiple actions we have the two invert 
    actions and the AND action and then the final invert actions. So we get many more actions in our agenda that also 
    will be executed sometimes at the same time. And when you have several items in the agenda that are executed at 
    the same time you can essentially get interleavings you can have some things that happen at some moment before 
    something else happened. And that was essentially here what we observed that at time 10 you got one item in the 
    agenda that just decided that the signal should be false and at the same time you got another item that put the 
    signal back to true. So essentially it's the fact that you have instead of a single item in the agenda you have 
    multiple items which are not executed atomically as a whole they get interspersed with each other and that causes 
    this flattering behavior where you see events that you didn't see before. So to summarize state and assignments 
    make our mental model of computation more complicated in particular because we lose referential transparency. On 
    the other hand assignments allow us to formulate certain programs in an elegant way. An example that we saw was 
    this great event simulation. Here a system is represented by a mutable list of actions and the effect of actions 
    when they're called are to change the state of objects and also to install other actions to be executed in the 
    future. As always the choice between functional and imperative programming must be made depending on the 
    situation. The digital circuit simulation was a good example for a mixture of functional and imperative 
    programming for essentially two reasons. One reason is that the non-hierarchical nature of circuit networks lends 
    itself well to an imperative formulation where we essentially place gates and circuits between wires. The other 
    reason is that we simulated a real world system where things change with the system where the internal state also 
    changes so this is quite natural."""


@pytest.fixture
def ocr_text():
    return """Advanced Example: Discrete Event Simulation
    We now consider an example of how assignments and higher-order
    functions can be combined in interesting ways.
    We will construct a digital circuit simulator.
    This example also shows how to build programs that do discrete event
    simulation."""


@pytest.fixture
def dirty_ocr_text():
    return """
        09:41 Tue 9 Jan
        * 100%
        < 88 Q D O
        formalisme de hamilton -
        monare acion et contrainte
        formalisme de hamilton
        O ITI
        x et 8 du type
        Paolo De Los Rios
        Lorsque
        uue relabiou eutre
        ou
        la relatiou
        est tiera lement comblée
        par
        8 = f"
        c'est à olice la fouchon iverse.
        La tansformatiou de Legenore est une façou torodue oe
        faire la meme chose, en passput par
        la derivée :
        dy
        et ou cher che
        glx)
        telle
        que
        y=
        de
        Pour tzouver gcx)
        procede aiusi :
        on substitue 8=8'(x)"""


@pytest.fixture
def slides_and_clean_concepts():
    return [
        {'number': 16, 'concepts': ['Antarctic ice sheet', 'Ice-sheet model', 'Arctic sea ice decline',
                                    'Polar ice cap', 'Arctic sea ice ecology and history', 'Greenland ice sheet',
                                    'Arctic ice pack', 'Ice sheet', 'Ice stream', 'Drift ice', 'Sea ice']},
        {'number': 38, 'concepts': []},
        {'number': 31, 'concepts': ['Climate model', 'General circulation model', 'Climate change']},
        {'number': 9, 'concepts': []},
        {'number': 2, 'concepts': []},
        {'number': 34, 'concepts': ['German language', 'German dialects', 'European integration', 'German studies',
                                    'Europe', 'Western Europe']},
        {'number': 24, 'concepts': ['Partial differential equation', 'Nonlinear system',
                                    'Euler equations (fluid dynamics)']}
    ]
